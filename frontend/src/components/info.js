import { 
  WRITE_PROMPT,
  WAIT_FOR_IMAGES,
  WAIT_FOR_PLAYERS,
  SELECT_IMAGE,
  END_OF_GAME,
  WAIT_FOR_START
} from '../constants/modes'
import { useSelector, shallowEqual } from 'react-redux'
import { styles } from '../styles'

const Info = () => {
  const mode = useSelector(state => state.mode, shallowEqual)
  const info = () => {
    switch(mode) {
      case WRITE_PROMPT:
        return 'Please write a prompt and click submit!'
      case WAIT_FOR_IMAGES:
        return 'Your images are being generated, sit back and relax...'
      case SELECT_IMAGE:
        return 'Your images have been generated, select the one that best matches your prompt.'
      case WAIT_FOR_PLAYERS:
        return 'Waiting for other players...'
      case END_OF_GAME:
        return 'That\'s the end of the game, have a look at the stories you made.'
      case WAIT_FOR_START:
        return 'Waiting for the admin to start the game.'
      default:
        return 'INSERT INSTRUCTION HERE'
    }
  }

  return (<div style={infoStyle}>{info()}</div>)
}

const infoStyle = {
  ...styles.singleLine,
  ...styles.text,
  paddingTop: '5vh',
  fontSize: '2vw',
  textAlign: 'center'
}
export default Info