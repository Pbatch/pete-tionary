import { LIGHT_BLUE, DARK_BLUE } from './constants/colours'

export const styles = {
  'text': {
    color: 'white',
    fontFamily: 'Arial'
  },
  'button': {
    color: 'white',
    fontFamily: 'Arial',
    backgroundColor: LIGHT_BLUE,
    borderRadius: '8px',
    ':hover': {
      backgroundColor: DARK_BLUE,
    }
  },
  'singleLine': {
    ':after': {
      content: "",
      display: 'inline-block',
      width: '100%',
    }
  },
  'arrow': {
    borderTop: '10px solid',
    borderRight: '10px solid',
    borderColor: LIGHT_BLUE,
    display: 'inline-block',
    padding: 'small',
    height: '1vw',
    width: '1vw',
    ':focus': {
      outlineStyle: 'none'
    },
    ':hover': {
      borderColor: DARK_BLUE
    }
  }
}
