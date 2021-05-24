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

  const images = state.urls.map((url) => {
    return (
      <img key={v4()} src={url} alt={url} onClick={handleClick} />
      )
    })

  const imageContainerStyle = {gridTemplateColumns: `repeat(${images.length}, 1fr)`}

  return (
    <div id='imageContainer' className='grid' style={imageContainerStyle}>
      {images}
    </div>
  )
}

export default Dream