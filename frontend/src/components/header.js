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
    <div></div>
    <div style={titleStyle}>
      <a href='https://pictionary.pbatch.net' style={{color: styles.font.color, textDecoration: 'none'}}>
        pictionary.pbatch.net
      </a>
    </div>
    <div></div>
    <div>
      <a href='https://www.buymeacoffee.com/pbatch' target='_blank' rel='noreferrer'>
        <img src={coffeeLogo} alt='Coffee logo'/>
      </a>
    </div>
  </div>
)

const gridStyle = {
  ...styles.font,
  gridTemplateColumns: 'repeat(5, 1fr)',
  display: 'grid',
  rowGap: '10px',
  columnGap: '10px',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: styles.button.backgroundColor
}

const titleStyle = {
  fontFamily: 'Helvetica Neue'
}

export default Header