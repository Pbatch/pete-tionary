import React, { useState, useEffect, useCallback } from 'react'
import Authenticator from './authenticator'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { AuthState, onAuthUIStateChange } from '@aws-amplify/ui-components'
import { SET_USERNAME } from '../constants/action-types'
import Router from './router'

const App = () => {
  const username = useSelector(state => state.username, shallowEqual)
  const dispatch = useDispatch()
  const setUsername = useCallback(
    (username) => dispatch({ type: SET_USERNAME, username }),
    [dispatch]
  )
  const [authState, setAuthState] = useState()

  useEffect(() => {
      return onAuthUIStateChange((nextAuthState, authData) => {
          setAuthState(nextAuthState)
          if (authData !== undefined) {
            setUsername(authData.username)
          }
      })
  }, [setUsername])

  return (authState === AuthState.SignedIn && username ? <Router /> : <Authenticator />)
}

export default App