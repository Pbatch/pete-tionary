import React, { useEffect, useState, useCallback } from 'react'
import { useDispatch } from 'react-redux'
import { API, graphqlOperation } from 'aws-amplify'
import { CreateRoom } from '../graphql/mutations'
import { OnCreateRoom } from '../graphql/subscriptions'
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
    syncRooms()
    let subscription = API.graphql(graphqlOperation(OnCreateRoom))
    .subscribe({
      next: () => syncRooms()
    })
    return () =>  {
      subscription.unsubscribe()
    }
  }, [setMode])

  async function syncRooms() {
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items
    setRooms(rooms)
  }

  async function createRoom() {
    const roomName = sampleSize(alphabet, 4).join('')
    const room = {'name': roomName }
    const roomData = await API.graphql(graphqlOperation(ListRooms))
    const rooms = roomData.data.listRooms.items
    const matchingRooms = rooms.filter((m) => m.name === room.name)
    if (matchingRooms.length === 0) {
      API.graphql(graphqlOperation(CreateRoom, room))
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