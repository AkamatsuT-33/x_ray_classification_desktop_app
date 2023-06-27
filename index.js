const electron = require('electron');
const st_app = electron.app;
const BrowserWindow = electron.BrowserWindow;
let mainWindow;

const subpy = null

st_app.on('ready', function() {
  global.subpy = require('child_process').spawn('python',['app.py']);
  let URL = 'http://localhost:5000';

  let openWindow = function() {
  mainWindow = new BrowserWindow({width: 1200, height: 800 });
  mainWindow.loadURL(URL);
  mainWindow.setMenuBarVisibility(false);
  };
  openWindow();
});

st_app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') 
    global.subpy.kill()
    st_app.quit()
})
