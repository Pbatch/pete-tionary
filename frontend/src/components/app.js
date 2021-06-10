import React, { useEffect } from 'react'
import Authenticator from './authenticator'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { Auth, Hub } from 'aws-amplify'
import { setUsername } from '../actions/index'
import Lobby from './lobby'
import Game from './game'

const App = () => {
  const state = useSelector(state => state, shallowEqual)
  const dispatch = useDispatch()

  useEffect(() => {
    let updateUser = async () => {
      try {
        let user = await Auth.currentAuthenticatedUser()
        dispatch(setUsername(user.username))
      } catch {
        dispatch(setUsername(''))
      }
    }
    updateUser()
    Hub.listen('auth', updateUser)
    return () => {
      Hub.remove('auth', updateUser)
    }
  }, [dispatch])

  if (state.username === '') {
    return <Authenticator />
  }
  else if (state.roomName !== '') {
    return <Game />
  }
  else {
    return <Lobby />
  }
}

export default App