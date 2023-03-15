import React from 'react';
import { Navbar, NavbarBrand, Nav, NavItem, NavLink } from 'reactstrap';
import './NavBar.css';

const NavBar = () => {
  return (
    <div>
      <Navbar color="light" light expand="md" className="navbar">
        <NavbarBrand href="/">Notebook Cell Predictor</NavbarBrand>
        <Nav className="ml-auto justify-content-end" navbar>
          <NavItem>
            <NavLink href="#">Menu</NavLink>
          </NavItem>
        </Nav>
      </Navbar>
    </div>
  );
};

export default NavBar;
