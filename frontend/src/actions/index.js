import { 
  SET_USERNAME,
  SET_URLS,
  SET_ROUND,
  SET_MODE
} from '../constants/action-types'

export const setUsername = (username) => ({
    type: SET_USERNAME,
    username
  })

export const setUrls = (urls) => ({
    type: SET_URLS,
    urls
  })

export const setRound = (round) => ({
    type: SET_ROUND,
    round
  })

export const setMode = (mode) => ({
  type: SET_MODE,
  mode
})