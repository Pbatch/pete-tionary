import { SET_URLS } from '../constants/action-types' 

const initialState = []

const Urls = (state = initialState, action) => {
  switch(action.type) {
    case SET_URLS:
      return action.urls
    default:
      return state
  }
}

export default Urls