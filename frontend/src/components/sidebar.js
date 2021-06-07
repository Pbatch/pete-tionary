import { v4 } from 'node-uuid'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import React, { useState, useEffect } from 'react'
import { ListMessages, ListRooms } from '../graphql/queries'
import { OnCreateMessage } from '../graphql/subscriptions'
import { DeleteRoom, DeleteMessage } from '../graphql/mutations'
import { END_OF_GAME, WAIT_FOR_START, SELECT_ROOM } from '../constants/modes'
import { setMode, setImages, setRound, setRoomName } from '../actions/index'

const Sidebar = () => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)
  const [usernames, setUsernames] = useState([])

  useEffect(() => {
    const fetchAndSetUsernames = async () => {
      const messageData = await API.graphql(graphqlOperation(ListMessages))
      const usernames = messageData.data.listMessages.items
      .filter(m => m.roomName === state.roomName && m.round === 0)
      .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))
      .map(m => m.username)
      setUsernames(usernames)
    }

    // As state.roomName is a dependency, the usernames are fetched when state.roomName changes
    fetchAndSetUsernames()

    // When a message is created, fetch the usernames
    let subscription = API.graphql(graphqlOperation(OnCreateMessage))
    .subscribe({
      next: () => fetchAndSetUsernames()
    })
    return () =>  {
      subscription.unsubscribe()
    }
  }, [state.roomName])

  async function handleStartSubmit(e) {
    e.preventDefault()
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items 
    const roomIds = rooms
    .filter(r => r.roomName === state.roomName)
    .map(r => r.id)

    if (roomIds.length > 0) {
      const payload = {'id': roomIds[0]}
      API.graphql(graphqlOperation(DeleteRoom, payload))
    }
  }

  async function handleLobbySubmit(e) {
    e.preventDefault()

    // Delete all of your messages
    const messageData = await API.graphql(graphqlOperation(ListMessages))
    const messages = messageData.data.listMessages.items
    .filter(m => m.username === state.username)
    messages.forEach(m => {
      const payload = {'id': m.id}
      API.graphql(graphqlOperation(DeleteMessage, payload))
    })

    dispatch(setMode(SELECT_ROOM))
    dispatch(setImages([]))
    dispatch(setRound(0))
    dispatch(setRoomName(''))
  }

  let usernameList = usernames.map((username) => {
    return (<div key={v4()}>{username}</div>)
  })

  const startButtonVisible = (state.admin === state.roomName) && (state.mode === WAIT_FOR_START)
  const startButton = (startButtonVisible 
    ? <button style={buttonStyle} onClick={handleStartSubmit}>Start</button> 
    : <div></div>)

  const lobbyButtonVisible = [END_OF_GAME, WAIT_FOR_START].includes(state.mode)
  const lobbyButton = (lobbyButtonVisible
    ? <button style={buttonStyle} onClick={handleLobbySubmit}>Lobby</button>
    : <div></div>
  )

  return (
    <div style={sidebarStyle}>
      <br/>
      <br/>
      <div>
        <div><b>Room</b></div>
        <div>{state.roomName}</div>
      </div>
      <br/>
      <div>
        <b>Players</b>
      </div>
      {usernameList}
      <br/>
      <div>{startButton}</div>
      <br/>
      <div>{lobbyButton}</div>
    </div>
  )
}

const sidebarStyle = {
  fontFamily: 'Courier New, monospace',
  textAlign: 'center',
  fontSize: '2em'
}

const buttonStyle = {
  width: '4em',
  fontSize: '1em'
}

export default Sidebar