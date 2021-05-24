import { SET_USERNAME } from '../constants/action-types' 

const initialState = ''

const Username = (state = initialState, action) => {
  switch(action.type) {
    case SET_USERNAME:
      return action.username
    default:
      return state
  }
}

export default Username