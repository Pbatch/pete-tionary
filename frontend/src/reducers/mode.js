import { SET_MODE } from '../constants/action-types' 
import { SELECT_ROOM } from '../constants/modes'

const initialState = SELECT_ROOM

const Mode = (state = initialState, action) => {
  switch(action.type) {
    case SET_MODE:
      return action.mode
    default:
      return state
  }
}

export default Mode