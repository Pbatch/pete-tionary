import { WRITE_PROMPT } from '../constants/modes'
import { useSelector, shallowEqual } from 'react-redux'
import { styles } from '../styles'
import Radium from 'radium'

const Form = ({handleSubmit, formError}) => {  
  const mode = useSelector(state => state.mode, shallowEqual)
  

  return (
    <div id='form'>
      <form onSubmit={handleSubmit} style={formStyle}>
        <input 
          type='text' 
          disabled={mode !== WRITE_PROMPT} 
          style={inputStyle}
          autoCorrect={'off'}
          autoCapitalize={'none'}
          spellCheck={'false'}
        />
        <button 
          type='submit'
          disabled={mode !== WRITE_PROMPT} 
          style={buttonStyle}
        >
          Submit
        </button>
      </form>
      <div style={errorStyle}>
        {formError}
      </div>
    </div>
  )
}

const formStyle = {
  ...styles.font,
  display: 'flex',
  justifyContent: 'center',
  columnGap: '1vw',
  paddingTop: '3vh'
}

const inputStyle = {
  width: '30vw',
  outline: 'none',
  height: '4.5vh'
}

const buttonStyle = {
  ...styles.button,
  width: '10vw',
  height: '5vh',
  fontSize: '2vw'
}

const errorStyle = {
  ...styles.text,
  textAlign: 'center',
  paddingTop: '1vh'
}

export default Radium(Form)