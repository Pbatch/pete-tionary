import { WRITE_PROMPT } from '../constants/modes'
import { useSelector, shallowEqual } from 'react-redux'
import { styles } from '../styles'
import Radium from 'radium'

const Form = ({handleSubmit}) => {  
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
    </div>
  )
}

const formStyle = {
  ...styles.font,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  margin: '1em',
  columnGap: '1em'
}

const inputStyle = {
  width: '30vw',
  outline: 'none'
}

const buttonStyle = {
  ...styles.button,
  width: '10vw'
}

export default Radium(Form)