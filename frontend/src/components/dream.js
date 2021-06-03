import { v4 } from 'node-uuid'
import { setImages, setMode } from '../actions/index'
import { useDispatch, useSelector, shallowEqual } from 'react-redux'
import { WAIT_FOR_PLAYERS, SELECT_IMAGE, END_OF_GAME } from '../constants/modes'
import { CreateMessage } from '../graphql/mutations'
import { API, graphqlOperation } from 'aws-amplify'

const Dream = () => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)

  async function handleClick(e) {
    e.preventDefault()
    if (state.mode !== SELECT_IMAGE) return
    const newImages = [{'url': e.target.src,
                        'username': state.username,
                        'prompt': state.prompt
                       }]
    const message = {url: e.target.src, 
                     round: state.round + 1, 
                     username: state.username, 
                     roomName: state.roomName}
    API.graphql(graphqlOperation(CreateMessage, message))
    dispatch(setImages(newImages))
    dispatch(setMode(WAIT_FOR_PLAYERS))
  }

  const imageStyle = {...imageStyle_, width: `${100/(state.images.length + 2)}vw`}

  const images = state.images.map(({ url, username, prompt }) => {
    const caption = (state.mode === END_OF_GAME) ? `${username}: "${prompt}"` : '' 
    return (
      <div key={v4()}>
        <img src={url} alt={url} onClick={handleClick} style={imageStyle} />
        <div style={captionStyle}>{caption}</div>
      </div>
      )
    })
 
  return (
    <div style={dreamStyle}>
      {images}
    </div>
  )
}

const imageStyle_ = {border: '1px solid black'}

const dreamStyle = {display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    columnGap: '25px',
                    margin: '10px',
                    minHeight: '30vw'}

const captionStyle = {padding: '10px',
                      fontFamily: 'Courier New, monospace',
                      textAlign: 'center',
                      fontSize: '2vw'}

export default Dream