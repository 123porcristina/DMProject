import pandas as pd

class Model:
    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.fileName = None
        self.fileContent = ""
        self.fileTotal = ""


    def isValid(self, fileName):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try:
            file = open(fileName, 'r')
            file.close()
            return True
        except:
            return False

    def setFileName(self, fileName):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''

        if self.isValid(fileName):
            self.fileName = fileName
            #self.fileContents = open(fileName, 'r', encoding="utf8").read()
            self.fileContents = pd.read_csv(self.fileName)
            self.fileTotal = str(self.fileContents.shape[0])

        else:
            self.fileContents = ""
            self.fileName = ""
            self.fileTotal = ""

    def getFileName(self):
        '''
        Returns the name of the file name member.
        '''
        return self.fileName

    def getFileContents(self):
        '''
        Returns the contents of the file if it exists, otherwise
        returns an empty string.
        '''

        return self.fileContents

    def getFileTotal(self):
        '''
        Returns the total of the dataframe if it has records, otherwise
        returns an empty string.
        '''

        return self.fileTotal



    def writeDoc(self, text):
        '''
        Writes the string that is passed as argument to a
        a text file with name equal to the name of the file
        that was read, plus the suffix ".bak"
        '''
        if self.isValid(self.fileName):
            fileName = self.fileName + ".bak"
            file = open(fileName, 'w')
            file.write(text)
            file.close()
