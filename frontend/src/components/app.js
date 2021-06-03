import React, { useState, useEffect } from 'react'
import Authenticator from './authenticator'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import { AuthState, onAuthUIStateChange } from '@aws-amplify/ui-components'
import { setUsername } from '../actions/index'
import Router from './router'

const App = () => {
  const username = useSelector(state => state.username, shallowEqual)
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

  return (authState === AuthState.SignedIn && username ? <Router /> : <Authenticator />)
}

export default App