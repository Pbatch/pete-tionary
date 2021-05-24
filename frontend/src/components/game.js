import React, { useEffect, useState, useCallback } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { LAMBDA_CONFIG } from '../constants/config.js'
import { CreateMessage } from '../graphql/mutations.js'
import { OnCreateMessage } from '../graphql/subscriptions.js'
import { ListMessages } from '../graphql/queries.js'
import { API, graphqlOperation } from 'aws-amplify'
import { SET_URLS, SET_ROUND, SET_MODE } from '../constants/action-types'
import { WAIT_FOR_IMAGES, SELECT_IMAGE, WRITE_PROMPT } from '../constants/modes'
import Form from './form.js'
import Dream from './dream.js'

const Game = () => {
  const state = useSelector(state => state, shallowEqual)
  const dispatch = useDispatch()
  const setUrls = useCallback(
    (urls) => dispatch({ type: SET_URLS, urls }),
    [dispatch]
  )
  const setRound = useCallback(
    (round) => dispatch({ type: SET_ROUND, round }),
    [dispatch]
  )
  const setMode = useCallback(
    (mode) => dispatch({ type: SET_MODE, mode }),
    [dispatch]
  )

  let subscription
  const [prompt, setPrompt] = useState('')

  console.log('state', state)

  const calculateUrl = (messages, nUsers, round, username) => {
    const roundEntries = messages.filter((message) => message.round === round) 
    console.log('round', round)
    console.log('round entries', roundEntries)
    const sortedEntries = roundEntries
    .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))
    const usernameIndex = sortedEntries.findIndex((m) => m.username === username)
    const urlIndex = ((usernameIndex + round) % nUsers + nUsers) % nUsers
    const url = sortedEntries[urlIndex].url
    return url
  }

  useEffect(() => {
    createMessage({url: '', username: state.username, round: 0})
    subscription = API.graphql(graphqlOperation(OnCreateMessage))
    .subscribe({
      next: async () => {
        const messageData = await API.graphql(graphqlOperation(ListMessages))
        const messages = messageData.data.listMessages.items 
        const nUsers = messages.filter((message) => message.round === 0).length
        const newRound = state.round + 1
        const currentRoundEntries = messages.filter((message) => message.round === newRound) 
        if (nUsers !== currentRoundEntries.length) return
        if (newRound === nUsers) {
          const newUrls = Array(nUsers)
          for (let round=0; round < nUsers; round++) {
            console.log('round', round)
            newUrls[round] = calculateUrl(messages, nUsers, round+1, state.username)
          }
          setUrls(newUrls)
          setPrompt('')
        }
        else {
          const newUrls = [calculateUrl(messages, nUsers, newRound, state.username)] 
          setUrls(newUrls)
          setRound(newRound)
          setPrompt('')
          setMode(WRITE_PROMPT)
        }
      }
    })
    return () =>  {
      subscription.unsubscribe()
    }
  }, [state])

  async function createMessage(message) {
    const messageData = await API.graphql(graphqlOperation(ListMessages))
    const messages = messageData.data.listMessages.items
    const matchingMessages = messages.filter((m) => m.round === message.round && m.username === message.username)
    if (matchingMessages.length === 0) {
      API.graphql(graphqlOperation(CreateMessage, message))
    }
  }
  
  function generateImages() {
    const payload = {'function': 'generate_images',
                     'jobDefinition': LAMBDA_CONFIG['jobDefinition'],
                     'jobQueue': LAMBDA_CONFIG['jobQueue'],
                     'bucket': LAMBDA_CONFIG['bucket'],
                     'prompt': prompt}
    fetch(LAMBDA_CONFIG['lambdaUrl'], {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  }

  async function checkImagesReady() {
    const payload = {'function': 'check_images_ready',
                     'bucket': LAMBDA_CONFIG['bucket'],
                     'prompt': prompt}
    let response = await fetch(LAMBDA_CONFIG['lambdaUrl'], {
      method: 'POST',
      body: JSON.stringify(payload)
    })
    return response.json()
  }

  async function waitForImages(){
    setMode(WAIT_FOR_IMAGES)
    let imagesReady = await checkImagesReady()
    if (imagesReady === 'not ready') {
      setTimeout(waitForImages, 10000)
    } else {
      const newUrls = Array(3)
      for (var i=0; i<3; i++) {
        newUrls[i] = `${LAMBDA_CONFIG['bucketUrl']}/prompt=${prompt}-seed=${i}.jpg`
      }
      setUrls(newUrls)
      setMode(SELECT_IMAGE)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const regex = /[a-zA-Z][a-zA-Z ]+/
    if (!regex.test(state.prompt)) return
    generateImages()
    waitForImages()
  }

  return (
    <div id='game'>
      <Dream 
        createMessage={createMessage}
        mode={state.mode}
        setMode={setMode}
      />
      <Form 
        mode={state.mode}
        prompt={prompt}
        setPrompt={setPrompt}
        handleSubmit={handleSubmit} 
       />
    </div>
  )
}

export default Game