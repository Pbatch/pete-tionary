import { v4 } from 'node-uuid'
import { setImages, setMode } from '../actions/index'
import { useDispatch, useSelector, shallowEqual } from 'react-redux'
import { WAIT_FOR_PLAYERS, SELECT_IMAGE, END_OF_GAME } from '../constants/modes'
import { CreateMessage } from '../graphql/mutations'
import { API, graphqlOperation } from 'aws-amplify'
import { useEffect, useState } from 'react'
import { styles } from '../styles'
import Radium from 'radium'
import { mod } from '../utils' 

const Dream = () => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)
  const [imageDisplay, setImageDisplay] = useState()
  const [visibility, setVisibility] = useState('hidden')
  const [offset, setOffset] = useState(0) 

  useEffect(() => {
    const handleClick = async (e) => {
      e.preventDefault()
      if (state.mode !== SELECT_IMAGE) return
      const newImages = state.images[0].filter((image) => image.url === e.target.src)
      const message = {url: e.target.src, 
                       round: state.round + 1, 
                       username: state.username, 
                       roomName: state.roomName}
      API.graphql(graphqlOperation(CreateMessage, message))
      dispatch(setImages([newImages]))
      dispatch(setMode(WAIT_FOR_PLAYERS))
    }
    // We have about 60vw to work with
    // This seems OK for 1-4 players, need to check 5+
    const imageWidth = Math.min(20, 60 / state.images.length)
    const imageStyle = {...imageStyle_, 
                        width: `${imageWidth}vw`}
    const captionStyle = {...captionStyle_,
                         width: `${imageWidth}vw`}
    
    const newImageDisplay = state.images[offset].map(({ url, username, prompt }) => {
      if (!prompt) return <div key={v4()}></div>
      const cleanPrompt = prompt.replaceAll(/_/g, " ")
      const captionText = `${username}: "${cleanPrompt}"`
      const caption = (state.mode === END_OF_GAME) ? captionText : '' 
      return (
        <div key={v4()}>
          <img src={url} alt={url} onClick={handleClick} style={imageStyle} />
          <div style={captionStyle}>{caption}</div>
        </div>
        )
      })
      setImageDisplay(newImageDisplay)
     
    setVisibility((state.mode === END_OF_GAME) ? 'visible' : 'hidden')

  }, [state, dispatch, offset])
 
  return (
    <div style={dreamStyle}>
      <div 
        key='leftArrow'
        style={{...leftArrowStyle, 'visibility': visibility}}
        onClick={() => setOffset(mod(offset - 1, state.images.length))} 
      />
      {imageDisplay}
      <div 
        key='rightArrow'
        style={{...rightArrowStyle, 'visibility': visibility}}
        onClick={() => setOffset(mod(offset + 1, state.images.length))} 
      />
    </div>
  )
}

const imageStyle_ = {border: '1px solid white'}

const dreamStyle = {display: 'flex',
                    columnGap: '3vw',
                    justifyContent: 'center',
                    minHeight: '55vh',
                    paddingTop: '10vh'}

const captionStyle_ = {...styles.text,
                       ...styles.singleLine,
                       textAlign: 'center',
                       paddingTop: '3vh',
                       fontSize: '1.5vw'}

const leftArrowStyle = {...styles.arrow,
                       'transform': 'rotate(-135deg'}

const rightArrowStyle = {...styles.arrow,
                        'transform': 'rotate(45deg)'}

export default Radium(Dream)