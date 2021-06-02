import React, { useEffect, useState, useCallback } from 'react'
import { useDispatch } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import { CreateRoom } from '../graphql/mutations'
import { OnCreateRoom, OnDeleteRoom } from '../graphql/subscriptions'
import { ListRooms } from '../graphql/queries'
import { sampleSize } from 'lodash'
import { v4 } from 'node-uuid'
import { Link, useHistory } from 'react-router-dom'
import { SET_ADMIN, SET_MODE } from '../constants/action-types'
import { SELECT_ROOM } from '../constants/modes'

const Rooms = () => {
  const history = useHistory()
  const dispatch = useDispatch()
  const setMode = useCallback(
    (mode) => dispatch({ type: SET_MODE, mode }),
    [dispatch]
  )
  const setAdmin = useCallback(
    (admin) => dispatch({ type: SET_ADMIN, admin }),
    [dispatch]
  )
  const [rooms, setRooms] = useState([])

  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

  useEffect(() => {
    setMode(SELECT_ROOM)
    setAdmin(false)
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
  }, [])

  async function fetchAndSetRooms() {
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items
    setRooms(rooms)
  }

  async function createRoom() {
    const roomName = sampleSize(alphabet, 4).join('')
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items
    const matchingRooms = rooms.filter((m) => m.name === roomName)

    if (matchingRooms.length === 0) {
      const payload = {'name': roomName, 'started': false}
      API.graphql(graphqlOperation(CreateRoom, payload))
      setAdmin(true)
      history.push(`/${roomName}`)
    }
  }

  const handleCreateRoomSubmit = (e) => {
    e.preventDefault()
    createRoom()
  }

  const roomNames = rooms.map(({name}) => {
    return (
      <div key={v4()}>
        <Link to={`/${name}`}>
          {name}
        </Link>
      </div>
    )
  })

  return (
    <div>
      {roomNames}
      <button onClick={handleCreateRoomSubmit}>Create Room</button>
    </div>
  )
}

export default Rooms