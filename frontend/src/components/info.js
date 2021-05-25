import { 
  WRITE_PROMPT,
  WAIT_FOR_IMAGES,
  WAIT_FOR_PLAYERS,
  SELECT_IMAGE,
  END_OF_GAME
} from '../constants/modes'

const Info = ({mode}) => {
  const info = () => {
    switch(mode) {
      case WRITE_PROMPT:
        return 'Please write a prompt and click submit!'
      case WAIT_FOR_IMAGES:
        return 'Your images are being generated. Sit back and relax...'
      case SELECT_IMAGE:
        return 'Your images have been generated. Please select the image that best matches your prompt.'
      case WAIT_FOR_PLAYERS:
        return 'Other players are yet to select an image. Tell them to hurry up!'
      case END_OF_GAME:
        return 'That\'s the end of the game, have a look at how your image evolved over time.'
      default:
        return 'INSERT INSTRUCTION HERE'
    }
  }

  return (<div>{info()}</div>)
}

export default Info