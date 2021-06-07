import React, { useEffect, useState } from 'react'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import { CreateRoom, CreateMessage } from '../graphql/mutations'
import { OnCreateRoom, OnDeleteRoom } from '../graphql/subscriptions'
import { ListRooms } from '../graphql/queries'
import { sampleSize } from 'lodash'
import { v4 } from 'node-uuid'
import { setMode, setAdmin, setImages, setRound, setRoomName } from '../actions'
import { SELECT_ROOM, WAIT_FOR_START } from '../constants/modes'

const Lobby = () => {
  const state = useSelector(state => state, shallowEqual)
  const dispatch = useDispatch()
  const [rooms, setRooms] = useState([])

  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

  useEffect(() => {  
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

  const fetchAndSetRooms = async () => {
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items
    setRooms(rooms)
  }

  const createRoom = async () => {
    // Warning! This causes disaster if the roomName is not unique
    const roomName = sampleSize(alphabet, 4).join('')
    const payload = {'roomName': roomName}
    API.graphql(graphqlOperation(CreateRoom, payload))
    dispatch(setAdmin(roomName))
    enterRoom(roomName)
  }

  const handleCreateRoomSubmit = (e) => {
    e.preventDefault()
    createRoom()
  }

  const handleEnterRoomSubmit = (e, roomName) => {
    e.preventDefault()
    enterRoom(roomName)
  }

  const enterRoom = async (roomName) => {       
    // Send an empty message to claim your space in the room
    const message = {url: '', round: 0, username: state.username, roomName: roomName}
    API.graphql(graphqlOperation(CreateMessage, message))

    // Set the initial state
    dispatch(setMode(WAIT_FOR_START))
    dispatch(setImages([]))
    dispatch(setRound(0))
    dispatch(setRoomName(roomName))
  }

  const roomButtons = rooms.map(({roomName}) => {
    return (
      <div key={v4()}>
        <button 
          onClick={(e) => handleEnterRoomSubmit(e, roomName)} 
          style={buttonStyle}
          disabled={state.mode !== SELECT_ROOM}
        >
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
        style={buttonStyle}
      >
        Create
      </button>
    </div>
  )

  const title = (
    <div style={titleStyle}>
      Rooms
    </div>
  )

  return (
    <div style={lobbyStyle}>
      {title}
      {roomButtons}
      {createRoomButton}
    </div>
  )
}

const titleStyle = {fontWeight: 'bold',
                    fontSize: '1em'}

const buttonStyle = {
  fontSize: '1em',
  margin: '1em'
}

const lobbyStyle = {textAlign: 'center', 
                   backgroundColor: 'white',
                   fontSize: '2em'}

export default Lobby