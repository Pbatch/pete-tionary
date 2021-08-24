import { SET_IMAGES } from '../constants/action-types' 

const initialState = [[]]

const Images = (state = initialState, action) => {
  switch(action.type) {
    case SET_IMAGES:
      return action.images
    default:
      return state
  }
}

export default Images