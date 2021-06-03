import Rooms from './rooms'
import Game from './game'
import { HashRouter, Route, Switch } from 'react-router-dom'

const Router = () => {
  return (
    <HashRouter>
      <Switch>
        <Route exact path='/'>
          <Rooms />
        </Route>
        <Route path='/:roomName'>
          <Game />
        </Route>
      </Switch>
    </HashRouter>
  )
}

export default Router