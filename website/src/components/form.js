function Form({handleSubmit, setPrompt}) {
  return (
    <div id='form'>
      <form onSubmit={handleSubmit}>
        <input id='input' type="text" onChange={e => setPrompt(e.target.value)} />
        <button id='button' type="submit">Submit</button>
      </form>
    </div>
  );
}

export default Form;