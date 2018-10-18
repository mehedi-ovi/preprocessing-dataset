# logging service Initialization
logger = logger()


def trailing_slash_reply(url, token):
    response = requests.get(url, headers=token)  # Get first page for count
    if response.status_code != 200:
        if response.status_code == 404 or response.status_code == 401:
            logger.info('Error status: {} {}'.format(str(response.status_code), "trying with trailing / ..."))
            response = requests.get(url + '/', headers=token)
            return response
        else:
            return logger.info('Error status: ' + str(response.status_code))
    return response


def getJsonData(url, token):
    response = trailing_slash_reply(url, token)

    json_data = response.json()
    logger.info("recieved data {} ".format(url))
    return json_data


def main():
    # Return all arguments in a list
    args = parser().parse_args()

    logger.info("Getting Access token.")
    token = GetAccessToken().getAccessToken(usertype=args.usertype, scopes=args.scopes)

    logger.info("Setup temp database to store requests to speed up restart download if network fails.")
    requests_cache.install_cache('requests_db', backend='sqlite')

    logger.info("Getting data with Access token.")
    json_data = getJsonData(args.url, token)
    logger.info(json_data)

    save_file(json_data, args.output_folder, args.filename)


def parser():
    parser = argparse.ArgumentParser(
        description=description)
    parser.add_argument('url',
                        type=str,
                        help='add full endpoint')
    parser.add_argument('usertype',
                        type=str,
                        choices=['employees'],
                        default='employees',
                        help='Choose user type')
    parser.add_argument('scopes',
                        type=str,
                        help='endpoints')
    parser.add_argument('out_folder',
                        type=str,
                        help='Add file location')
    parser.add_argument('filename',
                        type=str,
                        help='Add file-name')
    return parser


if __name__ == "__main__":
    main()
