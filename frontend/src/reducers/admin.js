import { SET_ADMIN} from '../constants/action-types' 

const initialState = false

const Admin = (state = initialState, action) => {
  switch(action.type) {
    case SET_ADMIN:
      return action.admin
    default:
      return state
  }
}

export default Admin