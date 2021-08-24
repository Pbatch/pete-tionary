import { styles } from '../styles'
import githubLogo from '../images/github.png'
import coffeeLogo from '../images/coffee.png'

const Header = () => (
  <div style={gridStyle}>
    <div>
      <a href='https://github.com/Pbatch/pictionary' target='_blank' rel='noreferrer'>
        <img src={githubLogo} alt='Github logo'/>
      </a>
    </div>
    <div style={titleStyle}>
      <a href='https://pictionary.pbatch.net' style={{color: styles.text.color, textDecoration: 'none'}}>
        Pete-tionary
      </a>
    </div>
    <div>
      <a href='https://www.buymeacoffee.com/pbatch' target='_blank' rel='noreferrer'>
        <img src={coffeeLogo} alt='Coffee logo'/>
      </a>
    </div>
  </div>
)

const gridStyle = {
  ...styles.font,
  gridTemplateColumns: 'repeat(3, 1fr)',
  display: 'grid',
  columnGap: '1vw',
  textAlign: 'center',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: styles.button.backgroundColor,
  height: '10vh'
}

const titleStyle = {
  fontFamily: 'Courier New Monospace',
  textShadow: '2px 2px black',
  fontSize: '4vw'
}

export default Header