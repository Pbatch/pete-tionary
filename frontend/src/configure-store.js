import { createStore } from 'redux'
import rootReducer from './reducers/index.js'
import { loadState, saveState } from './local-storage.js'
import throttle from 'lodash/throttle'

const Store = () => {
  const persistedState = loadState()
  const store = createStore(
    rootReducer,
    persistedState
  )
  
  store.subscribe(throttle(() => {
    saveState(store.getState())
  }, 1000))

  return (store)
}

export default Store