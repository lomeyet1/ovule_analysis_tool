# Automatically install missing packages
import subprocess
import sys

required_packages = ["PyQt6",'json','datetime']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


import json
import os
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QDialog, QLineEdit, QLabel, QCheckBox,
                             QHBoxLayout, QMessageBox, QScrollArea, QMenuBar,
                             QListWidget, QListWidgetItem, QFileDialog, QComboBox, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon


CONFIG_FILE = 'ovule_analysis_config.json'
BACKUP_DIR = 'backups'

class PathDialog(QDialog):
    def __init__(self, title, fields, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fields = fields
        self.config = config or {}
        self.values = {}
        self.param_widgets = []  # Stores widgets related to parameters
        self.setup_ui()

    def create_param_group(self, param=None):
        # Create parameter input group
        group = QVBoxLayout()

        # Parameter input row
        param_layout = QHBoxLayout()
        flag_input = QLineEdit(param.get('flag', '') if param else '')
        flag_input.setPlaceholderText("Parameter flag (e.g., --run)")
        value_input = QLineEdit(param.get('value', '') if param else '')
        value_input.setPlaceholderText("Parameter value")
        param_layout.addWidget(flag_input)
        param_layout.addWidget(value_input)

        # Add Browse and Default buttons
        button_layout = QHBoxLayout()
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda _: self.browse_param_path(value_input))
        default = QPushButton("Default")
        if param:
            # Save original value as Default. Only those created and successfully run in the design menu will be saved as Default
            param['default_value'] = param.get('value', '')
        default.clicked.connect(lambda _, v=value_input, p=param: self.set_default_value(v, p))
        button_layout.addWidget(browse)
        button_layout.addWidget(default)
        param_layout.addLayout(button_layout)

        group.addLayout(param_layout)

        # Parameter description row
        desc_layout = QHBoxLayout()
        desc_input = QLineEdit(param.get('description', '') if param else '')
        desc_input.setPlaceholderText("Parameter description (optional)")
        desc_layout.addWidget(desc_input)
        group.addLayout(desc_layout)

        # Checkbox row
        check_layout = QHBoxLayout()
        is_path_check = QCheckBox("Parameter value is a path (Check if the path exists before running)")
        is_fixed_check = QCheckBox("Fixed parameter (cannot be modified during runtime)")
        is_path_check.setChecked(param.get('is_path', False) if param else False)
        is_fixed_check.setChecked(param.get('is_fixed', False) if param else False)
        check_layout.addWidget(is_path_check)
        check_layout.addWidget(is_fixed_check)
        group.addLayout(check_layout)

        # Delete button
        remove_btn = QPushButton("Delete")
        remove_btn.clicked.connect(lambda: self.remove_param_group(group))
        check_layout.addWidget(remove_btn)

        # Store widget references
        self.param_widgets.append({
            'group': group,
            'flag': flag_input,
            'value': value_input,
            'description': desc_input,
            'is_path': is_path_check,
            'is_fixed': is_fixed_check,
            'remove': remove_btn
        })

        return group

    def set_default_value(self, value_input, param):
        """Set the default value of the parameter"""
        if param and 'default_value' in param:
            value_input.setText(param['default_value'])

    def browse_param_path(self, input_widget):
        """Provide file/directory browse functionality for parameter value"""
        # Pop up selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Browse Method")
        layout = QVBoxLayout()

        # Selection buttons
        file_btn = QPushButton("Select File")
        dir_btn = QPushButton("Select Directory")

        def choose_file():
            path = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.*)")
            if path[0]:
                input_widget.setText(path[0])
            dialog.accept()

        def choose_dir():
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
            if path:
                input_widget.setText(path)
            dialog.accept()

        file_btn.clicked.connect(choose_file)
        dir_btn.clicked.connect(choose_dir)

        layout.addWidget(file_btn)
        layout.addWidget(dir_btn)
        dialog.setLayout(layout)
        dialog.exec()

    def browse_path(self, field_id):
        """Provide file browse functionality for fields"""
        if field_id == 'carrier_path':
            file_filter = "Executable Files (*.exe);;All Files (*.*)" if sys.platform == 'win32' else "All Files (*.*)"
            path, _ = QFileDialog.getOpenFileName(self, "Select Carrier Program", "", file_filter)
        elif field_id == 'script_path':
            path, _ = QFileDialog.getOpenFileName(self, "Select Script File", "",
                                                  "Script Files (*.py *.ijm *.m);;All Files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Data Directory, Cancel to Select File") or \
                   QFileDialog.getOpenFileName(self, "Select File")[0]

        if path:
            self.inputs[field_id].setText(path)

    def remove_param_group(self, group):
        # Find and remove parameter group
        for i, widgets in enumerate(self.param_widgets):
            if widgets['group'] == group:
                # Remove all widgets from layout
                while group.count():
                    item = group.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                    elif item.layout():
                        while item.layout().count():
                            sub_item = item.layout().takeAt(0)
                            if sub_item.widget():
                                sub_item.widget().deleteLater()
                # Remove group from main layout
                self.param_layout.removeItem(group)
                # Remove reference from list
                self.param_widgets.pop(i)
                break

    def setup_ui(self):
        layout = QVBoxLayout()
        self.inputs = {}

        # Get existing categories and button names
        existing_categories = set()
        existing_buttons = {}  # {category: set(button_names)}
        for category, functions in self.config.items():
            if isinstance(functions, list):
                existing_categories.add(category)
                existing_buttons[category] = {func['button_name'] for func in functions}

        # Basic fields
        for field_id, field in self.fields.items():
            if field_id == 'parameters':
                continue  # Parameters section handled separately

            group = QVBoxLayout()
            group.addWidget(QLabel(field['label']))

            input_layout = QHBoxLayout()

            if field_id == 'category_name':
                combo = QComboBox()
                combo.setEditable(True)
                combo.addItems(sorted(existing_categories))
                combo.setCurrentText(field.get('value', ''))
                combo.currentTextChanged.connect(self.on_category_changed)
                self.inputs[field_id] = combo
                input_layout.addWidget(combo)

            elif field_id == 'button_name':
                combo = QComboBox()
                combo.setEditable(True)
                if field.get('value') and field.get('value') in self.fields.get('category_name', {}).get('value', ''):
                    category = self.fields['category_name']['value']
                    if category in existing_buttons:
                        combo.addItems(sorted(existing_buttons[category]))
                combo.setCurrentText(field.get('value', ''))
                self.inputs[field_id] = combo
                input_layout.addWidget(combo)

            else:
                text_input = QLineEdit(field.get('value', ''))
                self.inputs[field_id] = text_input
                input_layout.addWidget(text_input)

                if field.get('browse'):
                    browse = QPushButton("Browse")
                    browse.clicked.connect(lambda _, f=field_id: self.browse_path(f))
                    input_layout.addWidget(browse)

            if field.get('checkbox'):
                checkbox = QCheckBox(field['checkbox'])
                checkbox.setChecked(field.get('checked', False))
                self.inputs[f"{field_id}_check"] = checkbox
                group.addWidget(checkbox)

            group.addLayout(input_layout)
            layout.addLayout(group)

        # Set whether Add Parameter is needed
        if self.config.get('no_new_parameters_needed', False) == False:
            # Parameters section
            param_section = QGroupBox("Runtime Parameters")
            param_layout = QVBoxLayout()
            self.param_layout = param_layout  # Save reference for later Add Parameter group

            # Add existing parameters
            if 'parameters' in self.fields:
                for param in self.fields['parameters']:
                    param_layout.addLayout(self.create_param_group(param))

            # Add Parameter button
            add_param_btn = QPushButton("Add Parameter")
            add_param_btn.clicked.connect(lambda: param_layout.addLayout(self.create_param_group()))
            param_layout.addWidget(add_param_btn)

            param_section.setLayout(param_layout)
            layout.addWidget(param_section)

        self.config['no_new_parameters_needed'] = False  # Reset to show

        # ConfirmCancel buttons
        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        confirm = QPushButton("Confirm")
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

        self.setLayout(layout)

    def on_category_changed(self, category):
        if category in self.config and isinstance(self.config[category], list):
            button_combo = self.inputs['button_name']
            current_text = button_combo.currentText()
            button_combo.clear()
            button_combo.addItems(sorted({func['button_name'] for func in self.config[category]}))
            button_combo.setCurrentText(current_text)

    def get_values(self):
        values = {}
        # Get basic field values
        for k, v in self.inputs.items():
            if isinstance(v, (QLineEdit, QComboBox)):
                values[k] = v.currentText() if isinstance(v, QComboBox) else v.text()
            elif isinstance(v, QCheckBox):
                values[k] = v.isChecked()

        # Get parameter values
        parameters = []
        for widgets in self.param_widgets:
            if widgets['flag'].text() or widgets['value'].text():  # Only add non-empty parameters
                parameters.append({
                    'flag': widgets['flag'].text(),
                    'value': widgets['value'].text(),
                    'description': widgets['description'].text(),
                    'is_path': widgets['is_path'].isChecked(),
                    'is_fixed': widgets['is_fixed'].isChecked()
                })
        values['parameters'] = parameters

        return values


class ParamModifyDialog(QDialog):
    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func.copy()  # Create a copy of the function data
        self.param_inputs = []  # Store input widgets for modifiable parameters
        self.fixed_params = []  # Store fixed parameters
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Separate fixed parameters and modifiable parameters
        for param in self.func.get('parameters', []):
            param_copy = param.copy()  # Create a copy of the parameter
            if param.get('is_fixed'):
                self.fixed_params.append(param_copy)
            else:
                # Create UI for modifiable parameters
                group = QVBoxLayout()

                # Parameter input row
                input_layout = QHBoxLayout()
                label = QLabel(f"{param_copy['flag']}:")
                value_input = QLineEdit(param_copy['value'])
                input_layout.addWidget(label)
                input_layout.addWidget(value_input)

                # Add Browse and Default buttons
                button_layout = QHBoxLayout()
                browse = QPushButton("Browse")
                browse.clicked.connect(lambda _, w=value_input, p=param_copy: self.browse_path(w, p))

                default = QPushButton("Default")
                # Ensure default_value is correctly copied
                param_copy['default_value'] = param.get('default_value', param_copy['value'])
                default.clicked.connect(lambda _, w=value_input, p=param_copy: self.set_default_value(w, p))
                button_layout.addWidget(browse)
                button_layout.addWidget(default)
                input_layout.addLayout(button_layout)

                group.addLayout(input_layout)

                # Parameter description
                if param_copy.get('description'):
                    desc_label = QLabel(param_copy['description'])
                    desc_label.setStyleSheet("color: gray;")
                    group.addWidget(desc_label)

                layout.addLayout(group)
                self.param_inputs.append((param_copy, value_input))

        # Button area
        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        confirm = QPushButton("Confirm")
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

        self.setLayout(layout)
        self.setWindowTitle("Modify Parameters")

    def set_default_value(self, value_input, param):
        """Set the default value of the parameter"""
        if param and 'default_value' in param:
            value_input.setText(param['default_value'])


    def browse_path(self, input_widget, param):
        """Provide file/directory browse functionality for parameter value"""
        if param.get('is_path'):
            # If the parameter is a path type, filter by file extension
            file_filter = ""
            if param.get('description'):
                # Extract possible file type information from the description
                desc = param['description'].lower()
                if 'python' in desc or '.py' in desc:
                    file_filter = "Python Files (*.py);;All Files (*.*)"
                elif 'image' in desc or 'jpg' in desc or 'png' in desc:
                    file_filter = "Image Files (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*.*)"
                elif 'text' in desc or 'txt' in desc:
                    file_filter = "Text Files (*.txt);;All Files (*.*)"
                elif 'excel' in desc or 'xlsx' in desc or 'csv' in desc:
                    file_filter = "Data Files (*.xlsx *.xls *.csv);;All Files (*.*)"

            if not file_filter:
                file_filter = "All Files (*.*)"

            # Determine whether to select a file or directory based on the description
            if 'directory' in param.get('description', '').lower() or 'folder' in param.get('description',
                                                                                            '').lower() or 'dir' in param.get(
                'description', '').lower():
                path = QFileDialog.getExistingDirectory(self, "Select Directory")
            else:
                path = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)[0]
        else:
            # Non-path type, default to file selection
            path = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.*)")
            path = path[0]

        if path:
            input_widget.setText(path)

    def get_values(self):
        """Get all parameters, including fixed and modified parameters"""
        # Create a copy of the function data
        func_data = self.func.copy()

        # Update the values of modifiable parameters
        modified_params = []
        for param, input_widget in self.param_inputs:
            param_copy = param.copy()
            param_copy['value'] = input_widget.text()
            modified_params.append(param_copy)

        # Merge fixed parameters and modified parameters
        func_data['parameters'] = self.fixed_params + modified_params

        return func_data


class FeedbackDialog(QDialog):
    def __init__(self, feedback, params, all_fixed, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Runtime Feedback")
        self.feedback = feedback
        self.params = []  # Create parameter list
        # Ensure each parameter has a default_value
        for param in params:
            param_copy = param.copy()
            param_copy['default_value'] = param.get('default_value', param_copy['value'])
            self.params.append(param_copy)
        self.all_fixed = all_fixed
        self.param_inputs = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Add feedback information
        feedback_label = QLabel(self.feedback)
        feedback_label.setStyleSheet("font-size: 12pt;")  # Default color
        layout.addWidget(feedback_label)

        # Check if there are any non-fixed parameters
        show_param_group = any(not param.get('is_fixed') for param in self.params)

        if show_param_group:
            # Add Parameter display and modification area
            param_group = QGroupBox("Runtime Parameters")
            param_layout = QVBoxLayout()

            # Iterate through all parameters
            for param in self.params:
                # Display all parameters that are not marked as non-modifiable
                if not param.get('is_fixed'):  # Only display non-fixed parameters
                    # Parameter input row
                    input_layout = QHBoxLayout()

                    # Label and input box
                    label = QLabel(f"{param['flag']}:")
                    value_input = QLineEdit(param['value'])
                    input_layout.addWidget(label)
                    input_layout.addWidget(value_input)

                    # Add Browse and Default buttons
                    button_layout = QHBoxLayout()
                    browse = QPushButton("Browse")
                    browse.clicked.connect(lambda _, w=value_input, p=param: self.browse_path(w, p))
                    default = QPushButton("Default")
                    default.clicked.connect(lambda _, w=value_input, p=param: self.set_default_value(w, p))
                    button_layout.addWidget(browse)
                    button_layout.addWidget(default)
                    input_layout.addLayout(button_layout)

                    param_layout.addLayout(input_layout)

                    # Parameter description
                    if param.get('description'):
                        desc_label = QLabel(param['description'])
                        desc_label.setStyleSheet("color: gray;")
                        param_layout.addWidget(desc_label)

                    self.param_inputs.append((param, value_input))

            param_group.setLayout(param_layout)
            layout.addWidget(param_group)

        # Add buttons
        button_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(close_btn)

        # Always show the rerun button
        rerun_btn = QPushButton("Rerun")
        rerun_btn.clicked.connect(self.accept)
        button_layout.addWidget(rerun_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)


    def browse_path(self, input_widget, param):
        """Provide file/directory browse functionality for parameter value"""
        if param.get('is_path'):
            # If the parameter is a path type, filter by file extension
            file_filter = ""
            if param.get('description'):
                # Extract possible file type information from the description
                desc = param['description'].lower()
                if 'python' in desc or '.py' in desc:
                    file_filter = "Python Files (*.py);;All Files (*.*)"
                elif 'image' in desc or 'jpg' in desc or 'png' in desc:
                    file_filter = "Image Files (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*.*)"
                elif 'text' in desc or 'txt' in desc:
                    file_filter = "Text Files (*.txt);;All Files (*.*)"
                elif 'excel' in desc or 'xlsx' in desc or 'csv' in desc:
                    file_filter = "Data Files (*.xlsx *.xls *.csv);;All Files (*.*)"

            if not file_filter:
                file_filter = "All Files (*.*)"

            # Determine whether to select a file or directory based on the description
            if 'directory' in param.get('description', '').lower() or 'folder' in param.get('description', '').lower() or 'dir' in param.get('description', '').lower():
                path = QFileDialog.getExistingDirectory(self, "Select Directory")
            else:
                path = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)[0]
        else:
            # Non-path type, default to file selection
            path = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.*)")
            path = path[0]

        if path:
            input_widget.setText(path)

    def set_default_value(self, value_input, param):
        """set default value for the parameter"""
        if param and 'default_value' in param:
            value_input.setText(param['default_value'])

    def get_values(self):
        """get the modified Parameter value"""
        updated_params = []
        for param, input_widget in self.param_inputs:
            param_copy = param.copy()
            param_copy['value'] = input_widget.text()
            updated_params.append(param_copy)
        return updated_params


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ovule Analysis Tool')
        self.setWindowIcon(QIcon('ovule_icon.png'))  # Add icon
        self.config = self.load_config()
        self.setup_ui()
        self.setup_menu()

    def setup_menu(self):
        menubar = self.menuBar()
        for action, func in [
            ("Design", self.show_design_dialog),
            ("Update", self.show_update_dialog),
            ("Delete", self.show_delete_dialog),
            ("Backup", self.save_backup),
            ("Load", self.load_backup)
        ]:
            menubar.addAction(action).triggered.connect(func)

    def setup_ui(self):
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        self.setCentralWidget(scroll)

        layout = QVBoxLayout()
        self.load_existing_functions(layout)
        widget.setLayout(layout)

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def save_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def load_existing_functions(self, layout):
        for category, functions in self.config.items():
            if not isinstance(functions, list):
                continue
            layout.addWidget(QLabel(category))
            for func in functions:
                button = QPushButton(func['button_name'])
                button.clicked.connect(lambda _, f=func: self.run_function(f, show_confirm=False))
                layout.addWidget(button)

    def get_function_fields(self, values=None):
        v = values or {}
        return {
            'category_name': {'label': 'Category Name', 'value': v.get('category_name', '')},
            'button_name': {'label': 'Button Name', 'value': v.get('button_name', '')},
            'carrier_path': {
                'label': 'Carrier Path or Command',
                'value': v.get('carrier_path', ''),
                'browse': True,
                'checkbox': 'Use global command (e.g., python, matlab added to system environment variables)',
                'checked': v.get('use_global_command', False)
            },
            'script_path': {'label': 'Script Path', 'value': v.get('script_path', ''), 'browse': True},
            'parameters': v.get('parameters', []),
            'feedback': {'label': 'Runtime Feedback', 'value': v.get('feedback', 'Run Successful')}
        }

    def show_design_dialog(self, initial_data=None):
        dialog = PathDialog("Design Function", self.get_function_fields(initial_data), self.config, self)
        if dialog.exec():
            values = dialog.get_values()
            self.run_function({
                'category_name': values['category_name'],
                'button_name': values['button_name'],
                'carrier_path': values['carrier_path'],
                'script_path': values['script_path'],
                'use_global_command': values.get('carrier_path_check', False),
                'feedback': values.get('feedback', 'Run Successful'),
                'parameters': values['parameters']
            }, show_confirm=True)

    def show_update_dialog(self):
        # Show function selection list
        items = []
        for category, functions in self.config.items():
            if isinstance(functions, list):
                for func in functions:
                    items.append((f"{category} - {func['button_name']}",
                               (category, func)))

        if not items:
            QMessageBox.information(self, "Prompt", "No functions available for update")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Function to Update")
        layout = QVBoxLayout()
        list_widget = QListWidget()

        for text, data in items:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, data)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        confirm = QPushButton("Confirm")
        cancel.clicked.connect(dialog.reject)
        confirm.clicked.connect(dialog.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

        dialog.setLayout(layout)

        if dialog.exec():
            item = list_widget.currentItem()
            if item:
                category, func = item.data(Qt.ItemDataRole.UserRole)
                edit_dialog = PathDialog("Update Function", self.get_function_fields(func), self.config, self)
                if edit_dialog.exec():
                    values = edit_dialog.get_values()
                    new_func = {
                        'category_name': values['category_name'],
                        'button_name': values['button_name'],
                        'carrier_path': values['carrier_path'],
                        'script_path': values['script_path'],
                        'use_global_command': values.get('carrier_path_check', False),
                        'feedback': values.get('feedback', 'Run Successful'),
                        'parameters': values['parameters']
                    }

                    # Update config
                    functions = self.config[category]
                    for i, f in enumerate(functions):
                        if (f['button_name'] == func['button_name'] and
                            f['carrier_path'] == func.get('carrier_path', '') and
                            f['script_path'] == func.get('script_path', '')):
                            functions[i] = new_func
                            break
                    self.save_config()
                    self.setup_ui()

                    # Run updated function
                    self.run_function(new_func, show_confirm=True)

    def show_delete_dialog(self):
        items = []
        for category, functions in self.config.items():
            if isinstance(functions, list):
                for func in functions:
                    items.append((f"{category} - {func['button_name']}",
                               (category, func)))

        if not items:
            QMessageBox.information(self, "Prompt", "No functions available for delete")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Function to Delete")
        layout = QVBoxLayout()
        list_widget = QListWidget()

        for text, data in items:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, data)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        confirm = QPushButton("Confirm")
        cancel.clicked.connect(dialog.reject)
        confirm.clicked.connect(dialog.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

        dialog.setLayout(layout)

        if dialog.exec():
            item = list_widget.currentItem()
            if item:
                category, func = item.data(Qt.ItemDataRole.UserRole)
                reply = QMessageBox.question(
                    self, "Confirm Delete",
                    f"Are you sure you want to delete {category} - {func['button_name']}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    functions = self.config[category]
                    for i, f in enumerate(functions):
                        if (f['button_name'] == func['button_name'] and
                            f['carrier_path'] == func.get('carrier_path', '') and
                            f['script_path'] == func.get('script_path', '')):
                            del functions[i]
                            break
                    if not functions:
                        del self.config[category]
                    self.save_config()
                    self.setup_ui()

    def check_and_update_paths(self, func_data):
        while True:
            missing = {}
            # Check paths
            for path_type in ['carrier_path', 'script_path', 'data_path']:
                path = func_data.get(path_type)
                if not path:
                    continue

                # If carrier path uses global command, do not check if path exists
                if path_type == 'carrier_path' and func_data.get('use_global_command'):
                    continue

                if not os.path.exists(path):
                    missing[path_type] = {
                        'label': {'carrier_path': 'Carrier Path or Command',
                                'script_path': 'Script File',
                                'data_path': 'Data Path'}[path_type],
                        'value': path,
                        'browse': True
                    }
                    # If carrier path, add global command option
                    if path_type == 'carrier_path':
                        missing[path_type]['checkbox'] = 'Use global command (e.g., python, matlab added to system environment variables)'
                        missing[path_type]['checked'] = func_data.get('use_global_command', False)

            # Check paths in parameters
            for i, param in enumerate(func_data.get('parameters', [])):
                if param['is_path'] and param['value'] and not os.path.exists(param['value']):
                    param_id = f'param_{i}'
                    missing[param_id] = {
                        'label': f'Parameter Path ({param["description"] or param["flag"]})',
                        'value': param['value'],
                        'browse': True
                    }

            if not missing:
                return True

            self.config['no_new_parameters_needed'] = True
            dialog = PathDialog("Modify Invalid Paths", missing, self.config, self)
            if dialog.exec():
                values = dialog.get_values()

                if not values.get('parameters'):
                    values.pop('parameters', None)  # Remove empty parameters

                # Update parameter paths
                for k, v in values.items():
                    if k.startswith('param_'):
                        param_index = int(k.split('_')[1])
                        func_data['parameters'][param_index]['value'] = v

                values = {k: v for k, v in values.items() if not k.startswith('param_')}

                # Update other parts
                func_data.update(values)
                if 'carrier_path' in values:
                    func_data['use_global_command'] = values.get('carrier_path_check', False)

                # Update config file
                category = func_data['category_name']
                if category in self.config:
                    for f in self.config[category]:
                        if (f['button_name'] == func_data['button_name'] and
                            f['carrier_path'] == func_data.get('carrier_path', '') and
                            f['script_path'] == func_data.get('script_path', '')):
                            f.update(func_data)
                            break
                    self.save_config()
            else:
                return False

    def run_function(self, func_data, first_run=True, show_confirm=True):
        # Check paths
        if not self.check_and_update_paths(func_data):
            return

        # Get modifiable parameters
        modifiable_params = [p for p in func_data.get('parameters', []) if not p['is_fixed']]

        # 1. If there are modifiable parameters and it is the first run, show parameter modification dialog
        if modifiable_params and first_run:
            param_dialog = ParamModifyDialog(func_data, self)
            if param_dialog.exec() == QDialog.DialogCode.Accepted:
                # Update parameters
                func_data = param_dialog.get_values()
            elif show_confirm:
                # User clicked Cancel in parameter modification window, show design window again
                self.show_design_dialog(func_data)
            else:
                return

        # 2. Check input paths, invalid paths require update
        if not self.check_and_update_paths(func_data):
            return

        # 3. Get modifiable parameters
        modifiable_params = [p for p in func_data.get('parameters', []) if not p['is_fixed']]

        # 4. Initiate command
        cmd = []
        if func_data.get('use_global_command'):
            cmd.append(func_data['carrier_path'])
        else:
            cmd.append(os.path.abspath(func_data['carrier_path']))

        if func_data.get('script_path'):
            cmd.append(os.path.abspath(func_data['script_path']))

        # Add parameters
        for param in func_data.get('parameters', []):
            if param['flag'] and param['value']:   # Add flag only if both flag and value exist
                cmd.append(param['flag'])
            if param['value']:
                # If parameter is a path, convert to absolute path
                if param['is_path']:
                    cmd.append(os.path.abspath(param['value']))
                else:
                    cmd.append(param['value'])

        subprocess.Popen(cmd)

        # Update config file
        category = func_data['category_name']
        if category in self.config:
            for f in self.config[category]:
                if (f['button_name'] == func_data['button_name'] and
                    f['carrier_path'] == func_data.get('carrier_path', '') and
                    f['script_path'] == func_data.get('script_path', '')):
                    f.update(func_data)
                    break
            self.save_config()

        if func_data.get('feedback'):
            feedback_dialog = FeedbackDialog(func_data['feedback'], modifiable_params, self)
            if feedback_dialog.exec():
                # User clicked "Rerun"
                new_params = feedback_dialog.get_values()
                # Update parameters and rerun
                fixed_params = [p for p in func_data['parameters'] if p['is_fixed']]
                func_data['parameters'] = fixed_params + new_params
                self.run_function(func_data, first_run=False, show_confirm=show_confirm)
            elif show_confirm:
                # User clicked "Cancel" in feedback window
                # 6. Show confirmation window for forming button
                confirm_dialog = QDialog(self)
                confirm_dialog.setWindowTitle("Confirm Run Status")
                confirm_layout = QVBoxLayout()
                confirm_layout.addWidget(QLabel("Did the script run successfully?"))

                confirm_buttons = QHBoxLayout()
                return_btn = QPushButton("Return")
                confirm_btn = QPushButton("Confirm and Form Function")
                return_btn.clicked.connect(confirm_dialog.reject)
                confirm_btn.clicked.connect(confirm_dialog.accept)
                confirm_buttons.addWidget(return_btn)
                confirm_buttons.addWidget(confirm_btn)
                confirm_layout.addLayout(confirm_buttons)

                confirm_dialog.setLayout(confirm_layout)

                if confirm_dialog.exec() == QDialog.DialogCode.Accepted:
                    # Save function
                    category = func_data['category_name']
                    if category not in self.config:
                        self.config[category] = []

                    # Check for duplicate button names
                    existing_func = None
                    for i, f in enumerate(self.config[category]):
                        if (f['button_name'] == func_data['button_name']):
                            existing_func = (i, f)
                            break

                    if existing_func:
                        # If duplicate button name exists, update it
                        index, _ = existing_func
                        self.config[category][index] = func_data
                    else:
                        # If no duplicate button name, add new button
                        self.config[category].append(func_data)

                    self.save_config()
                    self.setup_ui()

                    show_confirm = False # Do not show subsequent confirmation windows

                    # Update config file
                    category = func_data['category_name']
                    if category in self.config:
                        for f in self.config[category]:
                            if (f['button_name'] == func_data['button_name'] and
                                f['carrier_path'] == func_data.get('carrier_path', '') and
                                f['script_path'] == func_data.get('script_path', '')):
                                f.update(func_data)
                                break
                        self.save_config()
                else:
                    # User clicked return in confirmation window, show design window again
                    self.show_design_dialog(func_data)  # Will cause show_confirm to become True

        elif show_confirm:
            # Without feedback window
            # 6. Show confirmation window for forming button
            confirm_dialog = QDialog(self)
            confirm_dialog.setWindowTitle("Confirm Run Status")
            confirm_layout = QVBoxLayout()
            confirm_layout.addWidget(QLabel("Was it successful?"))

            confirm_buttons = QHBoxLayout()
            return_btn = QPushButton("Return")
            confirm_btn = QPushButton("Confirm and Form Function")
            return_btn.clicked.connect(confirm_dialog.reject)
            confirm_btn.clicked.connect(confirm_dialog.accept)
            confirm_buttons.addWidget(return_btn)
            confirm_buttons.addWidget(confirm_btn)
            confirm_layout.addLayout(confirm_buttons)

            confirm_dialog.setLayout(confirm_layout)

            if confirm_dialog.exec() == QDialog.DialogCode.Accepted:
                # Save function
                category = func_data['category_name']
                if category not in self.config:
                    self.config[category] = []

                # Check for duplicate button names
                existing_func = None
                for i, f in enumerate(self.config[category]):
                    if (f['button_name'] == func_data['button_name']):
                        existing_func = (i, f)
                        break

                if existing_func:
                    # If duplicate button name exists, update it
                    index, _ = existing_func
                    self.config[category][index] = func_data
                else:
                    # If no duplicate button name, add new button
                    self.config[category].append(func_data)

                self.save_config()
                self.setup_ui()

                show_confirm = False # Do not show subsequent confirmation windows

                # Update config file
                category = func_data['category_name']
                if category in self.config:
                    for f in self.config[category]:
                        if (f['button_name'] == func_data['button_name'] and
                            f['carrier_path'] == func_data.get('carrier_path', '') and
                            f['script_path'] == func_data.get('script_path', '')):
                            f.update(func_data)
                            break
                    self.save_config()
            else:
                # User clicked return in confirmation window, show design window again
                self.show_design_dialog(func_data)  # Will cause show_confirm to become True

    def save_backup(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Backup",
            os.path.join(BACKUP_DIR, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
            "JSON files (*.json)"
        )
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "Success", "Backup saved")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save backup: {str(e)}")

    def load_backup(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Backup", BACKUP_DIR, "JSON files (*.json)"
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.save_config()
                self.setup_ui()
                QMessageBox.information(self, "Success", "Backup restored")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load backup: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    # window.resize(500, 200)
    window.show()
    sys.exit(app.exec())