import gql from 'graphql-tag';

export const CreateMessage = gql`
  mutation($url: String!, $username: String!, $round: Int!, $room: String!) {
    createMessage(input: {
      room: $room
      round: $round
      url: $url
      username: $username
    }) {
      id 
      room
      round
      url
      username
    }
  }
`

export const CreateRoom = gql`
  mutation($name: String!) {
    createRoom(input: {
      name: $name
    }) {
      id 
      name
    }
  }
`