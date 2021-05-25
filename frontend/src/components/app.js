import React, { useEffect, useState, useCallback } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { LAMBDA_CONFIG } from '../constants/config'
import { CreateMessage } from '../graphql/mutations'
import { OnCreateMessage } from '../graphql/subscriptions'
import { ListMessages } from '../graphql/queries'
import { API, graphqlOperation } from 'aws-amplify'
import { SET_URLS, SET_ROUND, SET_MODE } from '../constants/action-types'
import { WAIT_FOR_IMAGES, SELECT_IMAGE, WRITE_PROMPT, END_OF_GAME } from '../constants/modes'
import Form from './form'
import Dream from './dream'
import Info from './info'

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
  
  const mod = (a, b) => {
    return ((a % b) + b) % b
  }

  useEffect(() => {
    createMessage({url: '', username: state.username, round: 0})
    subscription = API.graphql(graphqlOperation(OnCreateMessage))
    .subscribe({
      next: async () => {
        const messageData = await API.graphql(graphqlOperation(ListMessages))
        const messages = messageData.data.listMessages.items 
        .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))
        const usernames = messages.filter((message) => message.round === 0).map(message => message.username)
        const nUsers = usernames.length
        const usernameIndex = usernames.indexOf(state.username)
        const newRound = state.round + 1
        const currentRoundEntries = messages.filter((message) => message.round === newRound) 

        // Not everyone's images are ready yet
        if (nUsers !== currentRoundEntries.length) return

        let newUrls
        if (newRound === nUsers) {
          // Get the usernames picture from the 1st round, 
          // the previous usernames picture from the 2nd round etc.
          
          newUrls = Array(nUsers)
          let roundEntries
          for (let round=1; round <= nUsers; round++) {
            roundEntries = messages.filter((message) => message.round === round)
            newUrls[round-1] = roundEntries[mod(usernameIndex - round + 1, nUsers)].url 
          }
          setMode(END_OF_GAME)
        }
        else {
          newUrls = [currentRoundEntries[mod(usernameIndex + 1, nUsers)].url]
          setRound(newRound)
          setMode(WRITE_PROMPT)
        }
        setUrls(newUrls)
        setPrompt('')
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
    <div style={appStyle}>
      <Info
        mode={state.mode}
      />
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
const appStyle = {textAlign: 'center', 
                  backgroundColor: 'white',
                  margin: '10px'}

export default Game