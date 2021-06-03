import { SET_ROOM_NAME } from '../constants/action-types' 

const initialState = ''

const RoomName = (state = initialState, action) => {
  switch(action.type) {
    case SET_ROOM_NAME:
      return action.roomName
    default:
      return state
  }
}

export default RoomName