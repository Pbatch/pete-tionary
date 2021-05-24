import React, { useState, useEffect, useCallback } from 'react'
import Authenticator from './authenticator.js'
import { useSelector, useDispatch, shallowEqual } from 'react-redux'
import Game from './game.js'
import { AuthState, onAuthUIStateChange } from '@aws-amplify/ui-components'
import { SET_USERNAME } from '../constants/action-types'

const Root = () => {
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

  return (authState === AuthState.SignedIn && username ? <Game /> : <Authenticator />)
}

export default Root