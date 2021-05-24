import { WRITE_PROMPT } from '../constants/modes'

const Form = ({mode, prompt, setPrompt, handleSubmit}) => {  
  function handleChange(e) {
    const newPrompt = e.target.value.replaceAll(" ", "_")
    setPrompt(newPrompt)
  }
  
  return (
    <div id='form'>
      <form onSubmit={handleSubmit}>
        <input id='input' type="text" disabled={mode !== WRITE_PROMPT} value={prompt} onChange={handleChange} />
        <button id='button' type="submit" disabled={mode !== WRITE_PROMPT}>Submit</button>
      </form>
    </div>
  )
}

export default Form