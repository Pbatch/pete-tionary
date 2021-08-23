import React, { useEffect } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { LAMBDA_CONFIG } from '../constants/config'
import { OnCreateMessage, OnDeleteRoom } from '../graphql/subscriptions'
import { ListMessages } from '../graphql/queries'
import { API, graphqlOperation } from 'aws-amplify'
import { setImages, setRound, setMode, setAdmin } from '../actions/index'
import { 
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

  const urlToPrompt = (url) => {
    // I have no idea how robust this function is
    // This used to be a regex but it didn't work in Safari
    // Example:
    // url = "https://s3-eu-west-1.amazonaws.com/pictionary-bucket/prompt=squidward-seed=1.jpg"
    // url.split('/').pop() => "prompt=squidward-seed=1.jpg"
    // url.split('/').pop().split('=')[1] => "squidward-seed"
    // url.split('/').pop().split('=')[1].split('-')[0] => "squidward"
    return url.split('/').pop().split('=')[1].split('-')[0]
  }
  
  const mod = (a, b) => {
    return ((a % b) + b) % b
  }  

  useEffect(() => {
    let onDeleteRoomPayload = {'roomName': state.roomName}
    let roomSubscription = API.graphql(graphqlOperation(OnDeleteRoom, onDeleteRoomPayload))
    .subscribe({
      next: async() => {
        dispatch(setMode(WRITE_PROMPT))
      }
    })

    let onCreateMessagePayload = {'roomName': state.roomName}
    let messageSubscription = API.graphql(graphqlOperation(OnCreateMessage, onCreateMessagePayload))
    .subscribe({
      next: async () => {
        const newRound = state.round + 1
        const nUsers = state.usernames.length
        const usernameIndex = state.usernames.indexOf(state.username)
        let payload = {'roomName': state.roomName, 'round': newRound}
        let messageData = await API.graphql(graphqlOperation(ListMessages, payload))
        let messages = messageData.data.listMessages.items
        .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))

        // Not everyone's images are ready yet
        if ((nUsers !== messages.length) || (nUsers === 0)) return
        
        let newImages = []
        if (newRound === nUsers) {
          // Fetch all the messages from the entire game
          let payload = {'roomName': state.roomName}
          let messageData = await API.graphql(graphqlOperation(ListMessages, payload))
          let messages = messageData.data.listMessages.items
          .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))

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
          let message = messages[mod(usernameIndex + 1, nUsers)]
          newImages.push({'url': message.url,
                          'username': message.username,
                          'prompt': urlToPrompt(message.url)
                         })
          dispatch(setRound(newRound))
          dispatch(setMode(WRITE_PROMPT))
        }
        dispatch(setImages(newImages))
      }
    })
    return () =>  {
      messageSubscription.unsubscribe()
      roomSubscription.unsubscribe()
    }
  }, [dispatch, state])
  
  function generateImages(prompt) {
    const payload = {'function': 'generate_images',
                     'queueUrl': LAMBDA_CONFIG['queueUrl'],
                     'bucket': LAMBDA_CONFIG['bucket'],
                     'prompt': prompt}
    fetch(LAMBDA_CONFIG['lambdaUrl'], {
      method: 'POST',
      body: JSON.stringify(payload)
    })
  }

  async function checkImagesReady(prompt) {
    const payload = {'function': 'check_images_ready',
                     'bucket': LAMBDA_CONFIG['bucket'],
                     'prompt': prompt}
    let response = await fetch(LAMBDA_CONFIG['lambdaUrl'], {
      method: 'POST',
      body: JSON.stringify(payload)
    })
    return response.json()
  }

  async function waitForImages(prompt){
    let imagesReady = await checkImagesReady(prompt)
    if (imagesReady === 'not ready') {
      setTimeout(waitForImages, 10000, prompt)
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

    // Replace all whitespace with a single space
    // Then replace the single spaces with underscores
    // Then make all characters lowercase
    const prompt = e.target[0].value.replace(/\s+/g, ' ').replace(/ /g, '_').toLowerCase()

    // All prompts must be 1 to 50 lowercase alphabet characters (+ underscores)
    const regex = /^[a-z_]{1,50}$/
    if (!regex.test(prompt)) return
    dispatch(setMode(WAIT_FOR_IMAGES))
    generateImages(prompt)
    waitForImages(prompt)
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
          handleSubmit={handleSubmit} 
        />
      </div>
    </div>
  )
}

const appStyle = {textAlign: 'center', 
                  margin: '1em',
                  display: 'flex'}

export default Game