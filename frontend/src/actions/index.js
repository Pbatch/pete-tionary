import { 
  SET_USERNAME,
  SET_IMAGES,
  SET_ROUND,
  SET_MODE,
  SET_ADMIN,
  SET_ROOM_NAME
} from '../constants/action-types'

export const setUsername = (username) => ({
    type: SET_USERNAME,
    username
  })

export const setImages = (images) => ({
    type: SET_IMAGES,
    images
  })

export const setRound = (round) => ({
    type: SET_ROUND,
    round
  })

export const setMode = (mode) => ({
  type: SET_MODE,
  mode
})

export const setAdmin = (admin) => ({
  type: SET_ADMIN,
  admin
})

export const setRoomName = (roomName) => ({
  type: SET_ROOM_NAME,
  roomName
})