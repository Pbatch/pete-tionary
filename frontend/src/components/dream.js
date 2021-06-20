import { v4 } from 'node-uuid'
import { setImages, setMode } from '../actions/index'
import { useDispatch, useSelector, shallowEqual } from 'react-redux'
import { WAIT_FOR_PLAYERS, SELECT_IMAGE, END_OF_GAME } from '../constants/modes'
import { CreateMessage } from '../graphql/mutations'
import { API, graphqlOperation } from 'aws-amplify'
import { useEffect, useState } from 'react'

const Dream = () => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)
  const [imageDisplay, setImageDisplay] = useState()

  useEffect(() => {
    const handleClick = async (e) => {
      e.preventDefault()
      if (state.mode !== SELECT_IMAGE) return
      const newImages = state.images.filter((image) => image.url === e.target.src)
      const message = {url: e.target.src, 
                       round: state.round + 1, 
                       username: state.username, 
                       roomName: state.roomName}
      API.graphql(graphqlOperation(CreateMessage, message))
      dispatch(setImages(newImages))
      dispatch(setMode(WAIT_FOR_PLAYERS))
    }
    const imageStyle = {...imageStyle_, width: `${80/(state.images.length + 2)}vw`}
    const newImageDisplay = state.images.map(({ url, username, prompt }) => {
      if (!prompt) return <div key={v4()}></div>
      const cleanPrompt = prompt.replaceAll(/_/g," ")
      const caption = (state.mode === END_OF_GAME) ? `${username}: "${cleanPrompt}"` : '' 
      return (
        <div key={v4()}>
          <img src={url} alt={url} onClick={handleClick} style={imageStyle} />
          <div style={captionStyle}>{caption}</div>
        </div>
        )
      })
      setImageDisplay(newImageDisplay)
  }, [state, dispatch])
 
  return (
    <div style={dreamStyle}>
      {imageDisplay}
    </div>
  )
}

const imageStyle_ = {border: '1px solid black'}

const dreamStyle = {display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    columnGap: '2em',
                    margin: '1em',
                    minHeight: '40vh'}

const captionStyle = {padding: '1em',
                      fontFamily: 'Courier New, monospace',
                      textAlign: 'center',
                      fontSize: '1em'}

export default Dream