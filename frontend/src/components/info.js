import { 
  WRITE_PROMPT,
  WAIT_FOR_IMAGES,
  WAIT_FOR_PLAYERS,
  SELECT_IMAGE,
  END_OF_GAME,
  WAIT_FOR_START
} from '../constants/modes'

const Info = ({mode}) => {
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
        return 'That\'s the end of the game, see how your image evolved over time.'
      case WAIT_FOR_START:
        return 'Waiting for the admin to start the game.'
      default:
        return 'INSERT INSTRUCTION HERE'
    }
  }

  return (<div style={infoStyle}>{info()}</div>)
}

const infoStyle = {
  fontFamily: 'Courier New, monospace',
  textAlign: 'center',
  fontSize: '25px'
}
export default Info