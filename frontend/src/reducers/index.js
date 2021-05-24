import { combineReducers } from "redux"
import round from './round'
import urls from './urls'
import username from './username'
import mode from './mode'

export default combineReducers({ 
  round,
  urls,
  username,
  mode
})
