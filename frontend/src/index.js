import React from 'react'
import ReactDOM from 'react-dom'
import App from './components/app'
import Amplify from 'aws-amplify'
import { AMPLIFY_CONFIG } from './constants/config'
import configureStore from './configure-store'
import { Provider } from 'react-redux'

Amplify.configure(AMPLIFY_CONFIG)
const store = configureStore()
document.body.style = 'background: #333; margin: 0'

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('root')
)
