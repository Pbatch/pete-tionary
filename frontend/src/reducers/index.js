import { combineReducers } from "redux"
import round from './round'
import images from './images'
import username from './username'
import mode from './mode'
import admin from './admin'

export default combineReducers({ 
  round,
  images,
  username,
  mode,
  admin
})