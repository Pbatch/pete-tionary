import { v4 } from 'node-uuid'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import React, { useEffect } from 'react'
import { ListMessages } from '../graphql/queries'
import { OnCreateMessage } from '../graphql/subscriptions'
import { DeleteRoom } from '../graphql/mutations'
import { END_OF_GAME, WAIT_FOR_START, SELECT_ROOM } from '../constants/modes'
import { setMode, setImages, setRound, setRoomName, setUsernames } from '../actions/index'
import { styles } from '../styles'
import Radium from 'radium'

const Sidebar = () => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)

  useEffect(() => {
    const fetchAndSetUsernames = async () => {
      const payload = {roomName: state.roomName, round: 0}
      const messageData = await API.graphql(graphqlOperation(ListMessages, payload))
      const usernames = messageData.data.listMessages.items
      .map(m => m.username)
      .sort()
      dispatch(setUsernames(usernames))
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
  }, [state.roomName, dispatch])

  async function handleStartSubmit(e) {
    e.preventDefault()

    // Delete the room so that no more players can join
    const payload = {'roomName': state.roomName}
    API.graphql(graphqlOperation(DeleteRoom, payload))
  }

  function handleLobbySubmit(e) {
    e.preventDefault()
    dispatch(setMode(SELECT_ROOM))
    dispatch(setImages([]))
    dispatch(setRound(0))
    dispatch(setRoomName(''))
    dispatch(setUsernames([]))
  }

  let usernameList = state.usernames.map((username) => {
    return (<div key={v4()}>{username}</div>)
  })

  // The start button is visible if you are the admin and waiting for the game to start
  const startButtonVisible = (state.admin === state.roomName) && (state.mode === WAIT_FOR_START)
  // The start button needs a key for hovering to work properly
  const startButton = (startButtonVisible 
    ? <button style={buttonStyle} onClick={handleStartSubmit} key={'startButton'}>Start</button> 
    : <div></div>)

  // You can only return to the lobby and the beginning or the end of the game
  const lobbyButtonVisible = [END_OF_GAME, WAIT_FOR_START].includes(state.mode)
  // The lobby button needs a key for hovering to work properly
  const lobbyButton = (lobbyButtonVisible
    ? <button style={buttonStyle} onClick={handleLobbySubmit} key={'lobbyButton'}>Lobby</button>
    : <div></div>
  )

  return (
    <div style={sidebarStyle}>
      <div style={{paddingTop: '15vh'}}>
        <div><b>Room</b></div>
        <div>{state.roomName}</div>
      </div>
      <div style={{paddingTop: '5vh'}}>
        <b>Players</b>
      </div>
      {usernameList}
      <div style={{paddingTop: '5vh'}}>
        {startButton}
      </div>
      <div style={{paddingTop: '3vh'}}>
        {lobbyButton}
      </div>
    </div>
  )
}

const sidebarStyle = {
  ...styles.text,
  backgroundColor: '#222222',
  fontSize: '2.5vw',
  textAlign: 'center'
}

const buttonStyle = {
  ...styles.button,
  fontSize: '2.5vw',
  width: '10vw'
}

export default Radium(Sidebar)