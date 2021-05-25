import { v4 } from 'node-uuid'
import { SET_URLS } from '../constants/action-types'
import { useCallback } from 'react'
import { useDispatch, useSelector, shallowEqual } from 'react-redux'
import { WAIT_FOR_PLAYERS, SELECT_IMAGE } from '../constants/modes'

const Dream = ({ createMessage, mode, setMode }) => {
  const dispatch = useDispatch()
  const state = useSelector(state => state, shallowEqual)
  const setUrls = useCallback(
    (urls) => dispatch({ type: SET_URLS, urls }),
    [dispatch]
  )

  async function handleClick(e) {
    e.preventDefault()
    if (mode !== SELECT_IMAGE) return
    const newUrl = e.target.src
    const message = {url: newUrl, username: state.username, round: state.round + 1}
    createMessage(message)
    setUrls([newUrl])
    setMode(WAIT_FOR_PLAYERS)
  }

  const imageStyle = {...imageStyle_, width: `${100/(state.urls.length + 2)}vw`}

  const images = state.urls.map((url) => {
    return (
      <div>
        <img key={v4()} src={url} alt={url} onClick={handleClick} style={imageStyle} />
        <div style={captionStyle}>INSERT CAPTION HERE</div>
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

const captionStyle = {padding: '10px'}

export default Dream