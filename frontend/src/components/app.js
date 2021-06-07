import React, { useState, useEffect } from 'react'
import Authenticator from './authenticator'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { AuthState, onAuthUIStateChange } from '@aws-amplify/ui-components'
import { setUsername } from '../actions/index'
import Lobby from './lobby'
import Game from './game'

const App = () => {
  const state = useSelector(state => state, shallowEqual)
  const dispatch = useDispatch()
  const [authState, setAuthState] = useState()

  useEffect(() => {
      return onAuthUIStateChange((nextAuthState, authData) => {
          setAuthState(nextAuthState)
          if (authData !== undefined) {
            dispatch(setUsername(authData.username))
          }
      })
  }, [dispatch, setAuthState])

  if (authState !== AuthState.SignedIn && !state.username) {
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