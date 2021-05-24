import { SET_MODE } from '../constants/action-types' 
import { WRITE_PROMPT } from '../constants/modes'

const initialState = WRITE_PROMPT

const Mode = (state = initialState, action) => {
  switch(action.type) {
    case SET_MODE:
      return action.mode
    default:
      return state
  }
}

export default Mode