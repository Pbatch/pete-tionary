import React, { useEffect, useState } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import { CreateRoom, CreateMessage, DeleteMessage } from '../graphql/mutations'
import { OnCreateRoom, OnDeleteRoom } from '../graphql/subscriptions'
import { ListRooms, ListMessages } from '../graphql/queries'
import { sampleSize } from 'lodash'
import { v4 } from 'node-uuid'
import { useHistory } from 'react-router-dom'
import { setMode, setAdmin, setImages, setRound, setRoomName } from '../actions'
import { SELECT_ROOM } from '../constants/modes'

const Rooms = () => {
  const state = useSelector(state => state, shallowEqual)
  const history = useHistory()
  const dispatch = useDispatch()
  const [rooms, setRooms] = useState([])

  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

  useEffect(() => {
    // On the initial render
    // Set mode to SELECT_ROOM, images to [] and round to 0
    // Fetch and set the rooms
    dispatch(setMode(SELECT_ROOM))
    dispatch(setImages([]))
    dispatch(setRound(0))
  
    fetchAndSetRooms()
    let createSubscription = API.graphql(graphqlOperation(OnCreateRoom))
    .subscribe({
      next: () => fetchAndSetRooms()
    })
    let deleteSubscription = API.graphql(graphqlOperation(OnDeleteRoom))
    .subscribe({
      next: () => fetchAndSetRooms()
    })
    return () =>  {
      createSubscription.unsubscribe()
      deleteSubscription.unsubscribe()
    }
  }, [dispatch])

  async function fetchAndSetRooms() {
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items
    setRooms(rooms)
  }

  async function createRoom() {
    // Warning! This causes disaster if the roomName is not unique
    const roomName = sampleSize(alphabet, 4).join('')
    const payload = {'roomName': roomName}
    API.graphql(graphqlOperation(CreateRoom, payload))
    enterRoom(roomName, roomName)
  }

  const handleCreateRoomSubmit = (e) => {
    e.preventDefault()
    createRoom()
  }

  const handleEnterRoomSubmit = (e, roomName) => {
    e.preventDefault()
    enterRoom(roomName, '')
  }

  const enterRoom = async (roomName, admin) => {    
    if (roomName !== state.roomName) {

      // Delete all your messages for other rooms
      const messageData = await API.graphql(graphqlOperation(ListMessages))
      const messages = messageData.data.listMessages.items
      .filter(m => m.username === state.username)
      messages.forEach(m => {
        const payload = {'id': m.id}
        API.graphql(graphqlOperation(DeleteMessage, payload))
      })
      
      // Send an empty message to claim your space in the room
      const message = {url: '', round: 0, username: state.username, roomName: roomName}
      API.graphql(graphqlOperation(CreateMessage, message))

      dispatch(setRoomName(roomName))
      dispatch(setAdmin(admin))
    }
    history.push(`/${roomName}`)
  }

  const roomButtons = rooms.map(({roomName}) => {
    return (
      <div key={v4()}>
        <button onClick={(e) => handleEnterRoomSubmit(e, roomName)}>
            {roomName}
        </button>
      </div>
    )
  })

  const createRoomButton = (
    <div>
      <button 
        onClick={handleCreateRoomSubmit}
        disabled={state.admin !== ''}
      >
        Create Room
      </button>
    </div>
  )

  return (
    <div style={roomStyle}>
      <b>Rooms</b>
      {roomButtons}
      {createRoomButton}
    </div>
  )
}

const roomStyle = {textAlign: 'center', 
                   backgroundColor: 'white',
                   margin: '10px',
                   fontSize: '25px'}

export default Rooms