import { v4 } from 'node-uuid'
import { useSelector, shallowEqual } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import React, { useState, useEffect, useRef } from 'react'
import { ListMessages, ListRooms } from '../graphql/queries'
import { OnCreateMessage } from '../graphql/subscriptions'
import { DeleteRoom } from '../graphql/mutations'
import { WAIT_FOR_START } from '../constants/modes'

const Sidebar = ({room}) => {
  const state = useSelector(state => state, shallowEqual)
  const [usernames, setUsernames] = useState([])

  async function fetchAndSetUsernames() {
    const messageData = await API.graphql(graphqlOperation(ListMessages))
    const usernames = messageData.data.listMessages.items
    .filter((message) => message.room === room && message.round === 0)
    .sort((m1, m2) => (m1.username > m2.username) - (m1.username < m2.username))
    .map((message) => message.username)
    setUsernames(usernames)
  }

  const mounted = useRef(false)
  useEffect(() => {
    if (!mounted.current) {
      fetchAndSetUsernames()
      mounted.current = true
    }
    let subscription = API.graphql(graphqlOperation(OnCreateMessage))
    .subscribe({
      next: () => fetchAndSetUsernames()
    })

    return () =>  {
      subscription.unsubscribe()
    }
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items 
    const roomIds = rooms
    .filter(r => r.name === room)
    .map(r => r.id)

    if (roomIds.length > 0) {
      const payload = {'id': roomIds[0]}
      API.graphql(graphqlOperation(DeleteRoom, payload))
    }
  }

  let usernameList = usernames.map((username) => {
    return (<div key={v4()}>{username}</div>)
  })

  const buttonVisible = (state.admin === true) && (state.mode === WAIT_FOR_START)
  const button = (buttonVisible ? <button style={buttonStyle} onClick={handleSubmit}>Start</button> : <div></div>)

  return (
    <div style={sidebarStyle}>
      <br/>
      <br/>
      <div>
        <div><b>Room</b></div>
        <div>{room}</div>
      </div>
      <br/>
      <div>
        <b>Players</b>
      </div>
      {usernameList}
      <br/>
      {button}
    </div>
  )
}

const sidebarStyle = {
  fontFamily: 'Courier New, monospace',
  textAlign: 'center',
  fontSize: '25px'
}

const buttonStyle = {
  width: '50%'
}

export default Sidebar