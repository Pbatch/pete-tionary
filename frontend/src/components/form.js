import { WRITE_PROMPT } from '../constants/modes'
import { useSelector, shallowEqual } from 'react-redux'

const Form = ({prompt, setPrompt, handleSubmit}) => {  
  const mode = useSelector(state => state.mode, shallowEqual)

  function handleChange(e) {
    const newPrompt = e.target.value.replace(/ /g, '_')
    setPrompt(newPrompt)
  }
  
  return (
    <div id='form'>
      <form onSubmit={handleSubmit} style={formStyle}>
        <input 
          type="text" 
          disabled={mode !== WRITE_PROMPT} 
          value={prompt.replace(/_/g, ' ')} 
          onChange={handleChange} 
          style={inputStyle}
        />
        <button 
          type="submit" 
          disabled={mode !== WRITE_PROMPT} 
          style={buttonStyle}
        >
          Submit
        </button>
      </form>
    </div>
  )
}

const formStyle = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  margin: '10px',
  columnGap: '10px'
}

const inputStyle = {
  width: '30vw'
}

const buttonStyle = {
  width: '10vw'
}

export default Form