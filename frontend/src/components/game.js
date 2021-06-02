import React, { useEffect, useState, useCallback, useRef } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { LAMBDA_CONFIG } from '../constants/config'
import { CreateMessage } from '../graphql/mutations'
import { OnCreateMessage, OnDeleteRoom } from '../graphql/subscriptions'
import { useLocation } from 'react-router-dom'
import { ListMessages, ListRooms } from '../graphql/queries'
import { API, graphqlOperation } from 'aws-amplify'
import { SET_IMAGES, SET_ROUND, SET_MODE } from '../constants/action-types'
import { WAIT_FOR_START, WAIT_FOR_IMAGES, SELECT_IMAGE, WRITE_PROMPT, END_OF_GAME } from '../constants/modes'
import Sidebar from './sidebar'
import Form from './form'
import Dream from './dream'
import Info from './info'

const Game = () => {
  const state = useSelector(state => state, shallowEqual)
  const dispatch = useDispatch()
  const setImages = useCallback(
    (images) => dispatch({ type: SET_IMAGES, images }),
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

  const [prompt, setPrompt] = useState('')
  const room = useLocation().pathname.replace(/\//g, '')

  const urlToPrompt = (url) => {
    const regex = /(?<==)(.*?)(?=-)/
    return regex.exec(url)[1].replace(/_/g, ' ')
  }
  
  const mod = (a, b) => {
    return ((a % b) + b) % b
  }  

  const mounted = useRef(false)
  useEffect(() => {
    if (!mounted.current) {
      console.log('Initial render')
      setMode(WAIT_FOR_START)
      setImages([])
      setRound(0)
      createMessage('', 0)
      mounted.current = true
    }
    let messageSubscription = API.graphql(graphqlOperation(OnCreateMessage))
    .subscribe({
      next: async () => {
        const messageData = await API.graphql(graphqlOperation(ListMessages))
        const messages = messageData.data.listMessages.items
        .filter((message) => message.room === room)
        .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))
        const usernames = messages.filter((message) => message.round === 0).map(message => message.username)
        const nUsers = usernames.length
        const usernameIndex = usernames.indexOf(state.username)
        const newRound = state.round + 1
        const currentRoundEntries = messages.filter((message) => message.round === newRound) 

        // Not everyone's images are ready yet
        if (nUsers !== currentRoundEntries.length) return
        
        let newImages = []
        if (newRound === nUsers) {
          // Get the usernames picture from the 1st round, 
          // the previous usernames picture from the 2nd round etc.
          for (let round=1; round <= nUsers; round++) {
            let roundEntries = messages.filter((message) => message.round === round)
            let message = roundEntries[mod(usernameIndex - round + 1, nUsers)]
            newImages.push({'url': message.url,
                            'username': message.username,
                            'prompt': urlToPrompt(message.url)
                            })
          }
          setMode(END_OF_GAME)
        }
        else {
          let message = currentRoundEntries[mod(usernameIndex + 1, nUsers)]
          newImages.push({'url': message.url,
                          'username': message.username,
                          'prompt': urlToPrompt(message.url)
                         })
          setRound(newRound)
          setPrompt('')
          setMode(WRITE_PROMPT)
        }
        setImages(newImages)
      }
    })
    let roomSubscription = API.graphql(graphqlOperation(OnDeleteRoom))
    .subscribe({
      next: async() => {
        const roomData = await API.graphql(graphqlOperation(ListRooms))
        const rooms = roomData.data.listRooms.items
        const roomStarted = !rooms.map((r) => r.name).includes(room)
        if (roomStarted) {
          setMode(WRITE_PROMPT)
        }
      }
    })
    return () =>  {
      messageSubscription.unsubscribe()
      roomSubscription.unsubscribe()
    }
  }, [])

  async function createMessage(url, round) {
    const message = {url: url, round: round, username: state.username, room: room}
    const messageData = await API.graphql(graphqlOperation(ListMessages))
    const messages = messageData.data.listMessages.items
    const matchingMessages = messages.filter((m) => m.round === message.round 
                                                    && m.username === message.username
                                                    && m.room === message.room
                                            )
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
    let imagesReady = await checkImagesReady()
    if (imagesReady === 'not ready') {
      setTimeout(waitForImages, 10000)
    } else {
      let newImages = []
      for (let i=0; i<3; i++) {
        newImages.push({'url': `${LAMBDA_CONFIG['bucketUrl']}/prompt=${prompt}-seed=${i}.jpg`,
                        'username': state.username,
                        'prompt': prompt
                      })
      }
      setMode(SELECT_IMAGE)
      setImages(newImages)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const regex = /[a-zA-Z][a-zA-Z ]+/
    if (!regex.test(state.prompt)) return
    setMode(WAIT_FOR_IMAGES)
    generateImages()
    waitForImages()
  }

  return (
    <div style={appStyle}>
      <div style={{width: '15vw'}}>
        <Sidebar room={room} />
      </div>
      <div style={{width: '85vw'}}>
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
    </div>
  )
}
const appStyle = {textAlign: 'center', 
                  backgroundColor: 'white',
                  margin: '10px',
                  display: 'flex'}

export default Game