import { SET_USERNAMES } from '../constants/action-types' 

const initialState = []

const Usernames = (state = initialState, action) => {
  switch(action.type) {
    case SET_USERNAMES:
      return action.usernames
    default:
      return state
  }
}

export default Usernames