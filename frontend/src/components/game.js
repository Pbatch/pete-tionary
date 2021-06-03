import React, { useEffect, useState } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { LAMBDA_CONFIG } from '../constants/config'
import { OnCreateMessage, OnDeleteRoom } from '../graphql/subscriptions'
import { ListMessages, ListRooms } from '../graphql/queries'
import { API, graphqlOperation } from 'aws-amplify'
import { setImages, setRound, setMode, setAdmin } from '../actions/index'
import { 
  WAIT_FOR_START, 
  WAIT_FOR_IMAGES, 
  SELECT_IMAGE, 
  WRITE_PROMPT, 
  END_OF_GAME 
} from '../constants/modes'
import Sidebar from './sidebar'
import Form from './form'
import Dream from './dream'
import Info from './info'

const Game = () => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)
  const [prompt, setPrompt] = useState('')

  const urlToPrompt = (url) => {
    const regex = /(?<==)(.*?)(?=-)/
    return regex.exec(url)[1].replace(/_/g, ' ')
  }
  
  const mod = (a, b) => {
    return ((a % b) + b) % b
  }  

  // On the initial load we need to:
  // Set the mode to WAIT_FOR_START
  // Clear all the state variables
  useEffect(() => {
    dispatch(setMode(WAIT_FOR_START))
    dispatch(setImages([]))
    dispatch(setRound(0))
  }, [dispatch])

  useEffect(() => {
    let roomSubscription = API.graphql(graphqlOperation(OnDeleteRoom))
    .subscribe({
      next: async() => {
        const roomData = await API.graphql(graphqlOperation(ListRooms))
        const rooms = roomData.data.listRooms.items
        const roomStarted = !rooms.map(r => r.roomName).includes(state.roomName)
        if (roomStarted) {
          dispatch(setMode(WRITE_PROMPT))
        }
      }
    })

    let messageSubscription = API.graphql(graphqlOperation(OnCreateMessage))
    .subscribe({
      next: async () => {
        const messageData = await API.graphql(graphqlOperation(ListMessages))
        const messages = messageData.data.listMessages.items
        .filter(m => m.roomName === state.roomName)
        .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))
        const usernames = messages
        .filter(m => m.round === 0)
        .map(m => m.username)
        const nUsers = usernames.length
        const usernameIndex = usernames.indexOf(state.username)
        const newRound = state.round + 1
        const currentRoundEntries = messages
        .filter(m => m.round === newRound) 

        // Not everyone's images are ready yet
        if (nUsers !== currentRoundEntries.length) return
        
        let newImages = []
        if (newRound === nUsers) {
          // Get the usernames picture from the 1st round, 
          // the previous usernames picture from the 2nd round etc.
          for (let round=1; round <= nUsers; round++) {
            let roundEntries = messages.filter(m => m.round === round)
            let message = roundEntries[mod(usernameIndex - round + 1, nUsers)]
            newImages.push({'url': message.url,
                            'username': message.username,
                            'prompt': urlToPrompt(message.url)
                            })
          }
          if (state.admin === state.roomName) {
            dispatch(setAdmin(''))
          }
          dispatch(setMode(END_OF_GAME))
        }
        else {
          let message = currentRoundEntries[mod(usernameIndex + 1, nUsers)]
          newImages.push({'url': message.url,
                          'username': message.username,
                          'prompt': urlToPrompt(message.url)
                         })
          dispatch(setRound(newRound))
          dispatch(setMode(WRITE_PROMPT))
          setPrompt('')
        }
        dispatch(setImages(newImages))
      }
    })
    return () =>  {
      messageSubscription.unsubscribe()
      roomSubscription.unsubscribe()
    }
  }, [dispatch, state])
  
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
      dispatch(setMode(SELECT_IMAGE))
      dispatch(setImages(newImages))
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const regex = /[a-zA-Z][a-zA-Z ]+/
    if (!regex.test(state.prompt)) return
    dispatch(setMode(WAIT_FOR_IMAGES))
    generateImages()
    waitForImages()
  }

  return (
    <div style={appStyle}>
      <div style={{width: '15vw'}}>
        <Sidebar />
      </div>
      <div style={{width: '85vw'}}>
        <Info />
        <Dream />
        <Form 
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