import { SET_ROUND } from '../constants/action-types' 

const initialState = 0

const Round = (state = initialState, action) => {
  switch(action.type) {
    case SET_ROUND:
      return action.round
    default:
      return state
  }
}

export default Round