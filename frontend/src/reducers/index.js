import { combineReducers } from "redux"
import round from './round'
import images from './images'
import username from './username'
import mode from './mode'
import admin from './admin'
import roomName from './roomName'
import usernames from './usernames'

export default combineReducers({ 
  round,
  images,
  username,
  mode,
  admin,
  roomName,
  usernames
})
